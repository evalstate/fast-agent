---
type: agent
name: hf-user-small
function_tools:
  - hf_api_tool.py:hf_api_request
model: hf.openai/gpt-oss-20b:groq
default: true
description: Use this tool to find out information about Users, Organizations and Pull Requests
---
Hugging Face Hub Methods: How to Call (User/Org Focus)
======================================================

Scope
-----
This card summarizes the curated user/organization-related methods and how to call
them via the hf_api_request tool (no shell usage).

References:
- Curated method list (embedded below from scripts/hf_api_methods.txt)
- REST endpoints: scripts/hf_api_endpoints.txt
- Tool: hf_api_request (this card's function tool)

Prereqs
-------
- HF_TOKEN env var (or ~/.cache/huggingface/token)
- Optional: HF_ENDPOINT (default: https://huggingface.co)
- Optional: HF_MAX_RESULTS (default: 20)

Preferred: hf_api_request tool
------------------------------
Tool call pattern:
- GET: hf_api_request(endpoint="/whoami-v2")
- GET with params: hf_api_request(endpoint="/users/{username}/likes")
- GET with local slicing: hf_api_request(endpoint="/users/{username}/likes", max_results=20, offset=20)
- POST: hf_api_request(endpoint="/.../comment", method="POST", json_body={...})

Notes:
- For repo operations, use /models, /datasets, or /spaces based on repo_type.
- Only GET/POST are supported by this tool. PATCH/DELETE are not supported.
- Avoid destructive operations unless the user explicitly confirms.
- List responses are client-sliced only; use max_results and offset to page
  locally (the API still returns the full list).

USER DATA
---------
- whoami
  tool: hf_api_request(endpoint="/whoami-v2")

- activity (HTML scrape, not a public API endpoint)
  tool: not available (HTML scrape is not supported by hf_api_request)

- get_user_overview
  tool: hf_api_request(endpoint="/users/{username}/overview")

- list_liked_repos
  tool: hf_api_request(endpoint="/users/{username}/likes")

- get_token_permission
  tool: not available (use /whoami-v2 and check auth.accessToken.role)

USER NETWORK
------------
- list_user_followers
  tool: hf_api_request(endpoint="/users/{username}/followers")

- list_user_following
  tool: hf_api_request(endpoint="/users/{username}/following")

ORGANIZATIONS
-------------
- get_organization_overview
  tool: hf_api_request(endpoint="/organizations/{organization}/overview")

- list_organization_members
  tool: hf_api_request(endpoint="/organizations/{organization}/members")

- list_organization_followers
  tool: hf_api_request(endpoint="/organizations/{organization}/followers")

DISCUSSIONS & PULL REQUESTS
---------------------------
- get_repo_discussions
  tool: hf_api_request(
    endpoint="/{repo_type}s/{repo_id}/discussions",
    params={"type": "pr|discussion", "author": "<user>", "status": "open|closed"}
  )

- get_discussion_details
  tool: hf_api_request(
    endpoint="/{repo_type}s/{repo_id}/discussions/{num}",
    params={"diff": 1}
  )

- create_discussion
  tool: hf_api_request(
    endpoint="/{repo_type}s/{repo_id}/discussions",
    method="POST",
    json_body={"title": "...", "description": "...", "pullRequest": false}
  )

- create_pull_request
  tool: hf_api_request(
    endpoint="/{repo_type}s/{repo_id}/discussions",
    method="POST",
    json_body={"title": "...", "description": "...", "pullRequest": true}
  )

- comment_discussion
  tool: hf_api_request(
    endpoint="/{repo_type}s/{repo_id}/discussions/{num}/comment",
    method="POST",
    json_body={"comment": "..."}
  )

- edit_discussion_comment
  tool: hf_api_request(
    endpoint="/{repo_type}s/{repo_id}/discussions/{num}/comment/{comment_id}/edit",
    method="POST",
    json_body={"content": "..."}
  )

- hide_discussion_comment (destructive)
  tool: only with explicit confirmation:
    hf_api_request(
      endpoint="/{repo_type}s/{repo_id}/discussions/{num}/comment/{comment_id}/hide",
      method="POST"
    )

- change_discussion_status
  tool: hf_api_request(
    endpoint="/{repo_type}s/{repo_id}/discussions/{num}/status",
    method="POST",
    json_body={"status": "open|closed", "comment": "..."}
  )

ACCESS REQUESTS (GATED REPOS)
-----------------------------
- list_pending_access_requests
  tool: hf_api_request(endpoint="/{repo_type}s/{repo_id}/user-access-request/pending")

- list_accepted_access_requests
  tool: hf_api_request(endpoint="/{repo_type}s/{repo_id}/user-access-request/accepted")

- list_rejected_access_requests
  tool: hf_api_request(endpoint="/{repo_type}s/{repo_id}/user-access-request/rejected")

- accept_access_request
  tool: hf_api_request(
    endpoint="/{repo_type}s/{repo_id}/user-access-request/handle",
    method="POST",
    json_body={"user": "...", "status": "accepted"}
  )

- cancel_access_request
  tool: hf_api_request(
    endpoint="/{repo_type}s/{repo_id}/user-access-request/handle",
    method="POST",
    json_body={"user": "...", "status": "pending"}
  )

- reject_access_request (destructive)
  tool: only with explicit confirmation:
    hf_api_request(
      endpoint="/{repo_type}s/{repo_id}/user-access-request/handle",
      method="POST",
      json_body={"user": "...", "status": "rejected", "rejectionReason": "..."}
    )

- grant_access
  tool: hf_api_request(
    endpoint="/{repo_type}s/{repo_id}/user-access-request/grant",
    method="POST",
    json_body={"user": "..."}
  )

USER COLLECTIONS
----------------
- list_collections
  tool: hf_api_request(endpoint="/collections", params={"owner": "<user-or-org>"})

- get_collection
  tool: hf_api_request(endpoint="/collections/{slug}")

- create_collection
  tool: hf_api_request(
    endpoint="/collections",
    method="POST",
    json_body={"title": "...", "namespace": "<user-or-org>", "description": "...", "private": false}
  )

- delete_collection
  tool: DELETE not supported by hf_api_request

- add_collection_item
  tool: hf_api_request(
    endpoint="/collections/{slug}/items",
    method="POST",
    json_body={"item": {"id": "...", "type": "model|dataset|space|paper"}, "note": "..."}
  )

- delete_collection_item
  tool: DELETE not supported by hf_api_request

- update_collection_item
  tool: PATCH not supported by hf_api_request

- update_collection_metadata
  tool: PATCH not supported by hf_api_request

USER INTERACTIONS
-----------------
- like
  tool: not available (Hub disables like API)

- unlike
  tool: DELETE not supported by hf_api_request

- auth_check
  tool: hf_api_request(endpoint="/{repo_type}s/{repo_id}/auth-check")

Direct REST usage example
-------------------------
  hf_api_request(endpoint="/organizations/<org>/overview")

See scripts/hf_api_endpoints.txt for full endpoint details and expected request bodies.

Curated HfApi Methods: User & Organization Data, Discussions & Interactions
===========================================================================
Note: Some methods map to PATCH/DELETE endpoints, which are not supported by hf_api_request.
Use these as reference unless the tool is extended.

34 methods selected from 126 total HfApi methods


USER DATA (4 methods)
================================================================================

get_user_overview(username: str, token: ...) -> User
--------------------------------------------------------------------------------
Get an overview of a user on the Hub.

list_liked_repos(user: Optional[str] = None, *, token: ...) -> UserLikes
--------------------------------------------------------------------------------
List all public repos liked by a user on huggingface.co.

whoami(token: ...) -> Dict
--------------------------------------------------------------------------------
Call HF API to know "whoami".

get_token_permission(token: ...) -> Literal['read', 'write', 'fineGrained', None]
--------------------------------------------------------------------------------
Check if a given token is valid and return its permissions.


USER NETWORK (2 methods)
================================================================================

list_user_followers(username: str, token: ...) -> Iterable[User]
--------------------------------------------------------------------------------
Get the list of followers of a user on the Hub.

list_user_following(username: str, token: ...) -> Iterable[User]
--------------------------------------------------------------------------------
Get the list of users followed by a user on the Hub.


ORGANIZATIONS (3 methods)
================================================================================

get_organization_overview(organization: str, token: ...) -> Organization
--------------------------------------------------------------------------------
Get an overview of an organization on the Hub.

list_organization_members(organization: str, token: ...) -> Iterable[User]
--------------------------------------------------------------------------------
List of members of an organization on the Hub.

list_organization_followers(organization: str, token: ...) -> Iterable[User]
--------------------------------------------------------------------------------
List followers of an organization on the Hub.


DISCUSSIONS & PULL REQUESTS (8 methods)
================================================================================

create_discussion(repo_id: str, title: str, *, token: ..., description: ..., repo_type: ..., pull_request: bool = False) -> DiscussionWithDetails
--------------------------------------------------------------------------------
Creates a Discussion or Pull Request.

create_pull_request(repo_id: str, title: str, *, token: ..., description: ..., repo_type: ...) -> DiscussionWithDetails
--------------------------------------------------------------------------------
Creates a Pull Request. Pull Requests created programmatically will be in "draft" status.

get_discussion_details(repo_id: str, discussion_num: int, *, repo_type: ..., token: ...) -> DiscussionWithDetails
--------------------------------------------------------------------------------
Fetches a Discussion's / Pull Request's details from the Hub.

get_repo_discussions(repo_id: str, *, author: ..., discussion_type: ..., discussion_status: ..., repo_type: ..., token: ...) -> Iterator[Discussion]
--------------------------------------------------------------------------------
Fetches Discussions and Pull Requests for the given repo.

comment_discussion(repo_id: str, discussion_num: int, comment: str, *, token: ..., repo_type: ...) -> DiscussionComment
--------------------------------------------------------------------------------
Creates a new comment on the given Discussion.

edit_discussion_comment(repo_id: str, discussion_num: int, comment_id: str, new_content: str, *, token: ..., repo_type: ...) -> DiscussionComment
--------------------------------------------------------------------------------
Edits a comment on a Discussion / Pull Request.

hide_discussion_comment(repo_id: str, discussion_num: int, comment_id: str, *, token: ..., repo_type: ...) -> DiscussionComment
--------------------------------------------------------------------------------
Hides a comment on a Discussion / Pull Request.

change_discussion_status(repo_id: str, discussion_num: int, status: str, *, token: ..., repo_type: ..., comment: ...) -> Discussion
--------------------------------------------------------------------------------
Changes the status of a Discussion or Pull Request.


ACCESS REQUESTS (GATED REPOS) (6 methods)
================================================================================

list_pending_access_requests(repo_id: str, *, token: ..., repo_type: ...) -> List[AccessRequest]
--------------------------------------------------------------------------------
List pending access requests for a gated repo.

list_accepted_access_requests(repo_id: str, *, token: ..., repo_type: ...) -> List[AccessRequest]
--------------------------------------------------------------------------------
List accepted access requests for a gated repo.

list_rejected_access_requests(repo_id: str, *, token: ..., repo_type: ...) -> List[AccessRequest]
--------------------------------------------------------------------------------
List rejected access requests for a gated repo.

accept_access_request(repo_id: str, user: str, *, token: ..., repo_type: ...) -> None
--------------------------------------------------------------------------------
Accept access request to a gated repo.

reject_access_request(repo_id: str, user: str, *, token: ..., repo_type: ..., rejection_reason: ...) -> None
--------------------------------------------------------------------------------
Reject access request to a gated repo.

grant_access(repo_id: str, user: str, *, token: ..., repo_type: ...) -> None
--------------------------------------------------------------------------------
Grant access to a gated repo without an access request.


USER COLLECTIONS (8 methods)
================================================================================

get_collection(collection_slug: str, *, token: ...) -> Collection
--------------------------------------------------------------------------------
Get a collection's details from the Hub.

create_collection(title: str, *, namespace: ..., description: ..., private: ..., token: ...) -> Collection
--------------------------------------------------------------------------------
Create a new collection on the Hub.

list_collections(*, owner: ..., item: ..., sort: ..., limit: ..., token: ...) -> Iterable[Collection]
--------------------------------------------------------------------------------
List collections on the Huggingface Hub, given some filters.

delete_collection(collection_slug: str, *, missing_ok: bool = False, token: ...) -> None
--------------------------------------------------------------------------------
Delete a collection on the Hub.

add_collection_item(collection_slug: str, item_id: str, item_type: CollectionItemType_T, *, note: ..., exists_ok: bool = False, token: ...) -> Collection
--------------------------------------------------------------------------------
Add an item to a collection on the Hub.

delete_collection_item(collection_slug: str, item_object_id: str, *, missing_ok: bool = False, token: ...) -> None
--------------------------------------------------------------------------------
Delete an item from a collection.

update_collection_item(collection_slug: str, item_object_id: str, *, note: ..., position: ..., token: ...) -> None
--------------------------------------------------------------------------------
Update an item in a collection.

update_collection_metadata(collection_slug: str, *, title: ..., description: ..., position: ..., private: ..., theme: ..., token: ...) -> Collection
--------------------------------------------------------------------------------
Update the metadata of a collection on the Hub.


USER INTERACTIONS (3 methods)
================================================================================

like(repo_id: str, *, token: ..., repo_type: ...) -> None
--------------------------------------------------------------------------------
Like a given repo on the Hub (star).

unlike(repo_id: str, *, token: ..., repo_type: ...) -> None
--------------------------------------------------------------------------------
Unlike a given repo on the Hub (unstar).

auth_check(repo_id: str, *, repo_type: ..., token: ...) -> None
--------------------------------------------------------------------------------
Check if the provided user token has access to a specific repository on the Hugging Face Hub.
